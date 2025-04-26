#include "matmul_transpose.h"
#include "cutlass/numeric_conversion.h"


using namespace cute;

template <typename T,
          typename SmemLayoutA,
          typename SmemLayoutB>
struct SharedStorage
{
    cute::ArrayEngine<T, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<T, cute::cosize_v<SmemLayoutB>> B;
};

template <typename T, int BM, int BK, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB, typename NumericConverter>
__global__ void mmt_kernel(const T *Aptr, T *Dptr, int m, int k)
{
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    if (ix > iy)
    {
        return;
    }

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); // non-owing copy of A
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, m), make_stride(m, Int<1>{}));
    Tensor DT = make_tensor(make_gmem_ptr(Dptr), make_shape(m, m), make_stride(Int<1>{}, m));  // non-owing copy of D transpose

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));  // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<BM>{}, Int<BK>{}), make_coord(ix, _));  // (BM, BK, num_tile_k)
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BM>{}), make_coord(iy, ix)); // (BM, BM)
    Tensor gDT = local_tile(DT, make_tile(Int<BM>{}, Int<BM>{}), make_coord(iy, ix)); // (BM, BM)

    // Compute tile residues for predication
    auto m_max_coord = m - size<0>(gA) * iy; 
    auto n_max_coord = m - size<0>(gB) * ix; 
    auto k_residue = k - size<1>(gA) * (size<2>(gA) - 1);
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Shared memory buffers
    // use char type to avoid issues with different template param T
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, SmemLayoutA, SmemLayoutB>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{}); 

    // register, use tiled_mma to partition register A/B/C
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCgD = thr_mma.partition_C(gD); 
    auto tCgDT = thr_mma.partition_C(gDT); // partition for D transposed

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
    clear(tCrD);

    // from global memory to shared memory
    G2SCopyA g2s_tiled_copy_a;
    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

    // from shared memory to register, use tiled_mma to generate tiled_copy
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    //
    // PREDICATES
    //
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA_copy), size<2>(tAsA_copy)), Stride<_1, _0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB_copy), size<2>(tBsB_copy)), Stride<_1, _0>{});

    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));

    Tensor tAcA = g2s_thr_copy_a.partition_S(cA);
    Tensor tBcB = g2s_thr_copy_b.partition_S(cB);

    // Set Predicates for row bounds
    CUTE_UNROLL
    for (int i = 0; i < size<0>(tApA); ++i)
    {
        tApA(i, 0) = get<0>(tAcA(0, i, 0)) < m_max_coord; // blk_m coord < m_max_coord
    }
    // Set predicates for col bounds
    CUTE_UNROLL
    for (int i = 0; i < size<0>(tBpB); ++i)
    {
        tBpB(i, 0) = get<0>(tBcB(0, i, 0)) < n_max_coord; // blk_n coord < n_max_coord
    }

    // Clear the smem tiles to account for predicated off loads
    clear(tAsA_copy);
    clear(tBsB_copy);

    Tensor tAgAk = tAgA_copy(_, _, _, size<2>(gA) - 1);
    CUTE_UNROLL
    for (int i = 0; i < size<2>(tAsA_copy); ++i)
    {
        if (get<1>(tAcA(0, 0, i)) < get<2>(residue_mnk))
        { // blk_k coord < residue_k (gA shifted)
            cute::copy_if(g2s_tiled_copy_a, tApA(_, i), tAgAk(_, _, i), tAsA_copy(_, _, i));
        }
    }

    Tensor tBgBk = tBgB_copy(_, _, _, size<2>(gB) - 1);
    CUTE_UNROLL
    for (int i = 0; i < size<2>(tBsB_copy); ++i)
    {
        if (get<1>(tBcB(0, 0, i)) < get<2>(residue_mnk))
        { // blk_k coord < residue_k (gA shifted)
            cute::copy_if(g2s_tiled_copy_b, tBpB(_, i), tBgBk(_, _, i), tBsB_copy(_, _, i));
        }
    }
    cp_async_fence();

    // loop over k: i. load tile, ii. mma
    int ntile = (k + BK - 1) / BK;
    CUTE_UNROLL
    for (int itile = 0; itile < ntile; ++itile)
    {
        if (itile >= 1)
        {
            // copy  (CPY, CPY_M, CPY_K) , async
            cute::copy_if(g2s_tiled_copy_a, tApA, tAgA_copy(_, _, _, itile - 1), tAsA_copy(_, _, _));
            cute::copy_if(g2s_tiled_copy_b, tBpB, tBgB_copy(_, _, _, itile - 1), tBsB_copy(_, _, _));
            cp_async_fence();
        }

        cp_async_wait<0>();
        __syncthreads();

        int nk = size<2>(tCrA);
        CUTE_UNROLL
        for (int ik = 0; ik < nk; ++ik)
        {
            // copy  (CPY, CPY_M), sync
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik), tCrA_view(_, _, ik));
            // copy  (CPY, CPY_N)
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik), tCrB_view(_, _, ik));
            // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // for ik

    } // itile

    __syncthreads();

    Tensor cD = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));
    Tensor tCcD = thr_mma.partition_C(cD);
    NumericConverter converter;
    CUTE_UNROLL
    for (int i = 0; i < size(tCrD); ++i)
    {
        if (elem_less(tCcD(i), make_coord(m_max_coord, n_max_coord)))
        {
            tCgD(i) = converter(tCrD(i));
        }
    }

    // copy for upper triangular parts.
    if (ix < iy)
    {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrD); ++i)
        {
            if (elem_less(tCcD(i), make_coord(m_max_coord, n_max_coord)))
            {
                tCgDT(i) = converter(tCrD(i));
            }
        }
    }
}

template <typename T>
void launch_mmt_kernel(T *x, T *y, int M, int K, cudaStream_t stream)
{
    auto BM = Int<128>{};
    auto BK = Int< 32>{};
    // Define the smem layouts
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<BM>{}, Int<BK>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<BM>{}, Int<BK>{}))); // (m,n) -> smem_idx

    // mma
    using mma_op = cute::conditional_t<
        cute::is_same_v<T, cute::half_t>,
        SM80_16x8x16_F32F16F16F32_TN,
        cute::conditional_t<
            cute::is_same_v<T, cute::bfloat16_t>,
            SM80_16x8x16_F32BF16BF16F32_TN,
            void 
            >>;
    static_assert(!cute::is_same_v<mma_op, void>,
                  "Unsupported data type. Expected cute::half_t or cute::bfloat16_t.");

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = typename mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 2 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), // Thr layout 32x4 k-major
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    using NumericConverter = cutlass::NumericConverter<T, float, cutlass::FloatRoundStyle::round_to_nearest>;

    int BX = (M + BM - 1) / BM;
    int BY = (M + BM - 1) / BM;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY);

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize =
        shm_size_AB * sizeof(T);

    int shm_size = kShmSize;

    auto kernel_fptr = mmt_kernel<T, BM, BK, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, S2RCopyAtomA, S2RCopyAtomB, NumericConverter>;
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<grid, block, shm_size>>>(x, y, M, K);
}

template void launch_mmt_kernel<cute::half_t>(cute::half_t *x, cute::half_t *y, int M, int K, cudaStream_t stream);
template void launch_mmt_kernel<cute::bfloat16_t>(cute::bfloat16_t *x, cute::bfloat16_t *y, int M, int K, cudaStream_t stream);