    if constexpr (STAGE) return ::stage_store<C>(a, m0, o.n0, acc);  // y in write-combined host memory
// The same kernels storing their fp32 tile as whole lines (stage_store), for a y in write-combined host memory.
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_st_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kChained, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_exact_st_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kExact, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_st_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kChained, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_exact_st_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kExact, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_st_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kChained, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true>(a); }