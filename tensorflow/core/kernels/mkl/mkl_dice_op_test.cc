#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)

#ifdef INTEL_MKL

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"


namespace tensorflow {
namespace {

enum class Device { CPU };

class MklDiceTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, int axis,
                          float epsilon) {
    TF_EXPECT_OK(NodeDefBuilder("dice", "_MklDice")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

//----------------------------------------------------------------------------//
//                            Accuracy Check                                  //
//----------------------------------------------------------------------------//
TEST_F(MklDiceTest, Dice_Test) {
  const int rows = 107;
  const int cols = 255;

  MakeOpAndSetDevice(Device::CPU, DT_FLOAT, 0, 1e-12);

  AddInput<float>(TensorShape({rows, cols}),
                  [](int i) -> float { return 2.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 1.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return -2.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 0.4; });

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({rows, cols}));
    float output_array[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
      output_array[i] = 1.4;
    }
    test::FillValues<float>(&expected_output, output_array);
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-5);
  }
}

//----------------------------------------------------------------------------//
//                       Performance benchmarks                               //
//----------------------------------------------------------------------------//
static Graph* DiceGraph(const string& kind, int rows, int cols) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType dtype = DT_FLOAT;

  Tensor in(dtype, TensorShape({rows, cols}));
  in.flat<float>().setRandom();
  Tensor rvar(dtype, TensorShape({cols}));
  rvar.flat<float>().setRandom();
  Tensor mean_rvar(dtype, TensorShape({cols}));
  mean_rvar.flat<float>().setRandom();
  Tensor alpha(dtype, TensorShape({cols}));
  alpha.flat<float>().setRandom();
  Tensor sub_const(dtype, TensorShape({1}));
  sub_const.flat<float>().setValues({1.0});
  const bool isDefault = (kind == "Default");

  Node* input_in = test::graph::Constant(g, in);
  Node* input_rvar = test::graph::Constant(g, rvar);
  Node* input_mean_rvar = test::graph::Constant(g, mean_rvar);
  Node* input_alpha = test::graph::Constant(g, alpha);
  Node* input_sub = test::graph::Constant(g, sub_const);

  Node* mul_0;
  Node* addv2_0;
  Node* sigmoid;
  Node* mul_1;
  Node* sub;
  Node* mul_2;
  Node* mul_3;
  Node* addv2_1;
  Node* dice;

  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(g->NewName("mul_0"), "Mul")
                    .Input(input_in)
                    .Input(input_rvar)
                    .Attr("T", dtype)
                    .Finalize(g, &mul_0));

    TF_CHECK_OK(NodeBuilder(g->NewName("addv2_0"), "AddV2")
                    .Input(mul_0)
                    .Input(input_mean_rvar)
                    .Attr("T", dtype)
                    .Finalize(g, &addv2_0));

    TF_CHECK_OK(NodeBuilder(g->NewName("sigmoid"), "Sigmoid")
                    .Input(addv2_0)
                    .Attr("T", dtype)
                    .Finalize(g, &sigmoid));

    TF_CHECK_OK(NodeBuilder(g->NewName("mul_1"), "Mul")
                    .Input(sigmoid)
                    .Input(input_in)
                    .Attr("T", dtype)
                    .Finalize(g, &mul_1));

    TF_CHECK_OK(NodeBuilder(g->NewName("sub"), "Sub")
                    .Input(input_sub)
                    .Input(sigmoid)
                    .Attr("T", dtype)
                    .Finalize(g, &sub));

    TF_CHECK_OK(NodeBuilder(g->NewName("mul_2"), "Mul")
                    .Input(sigmoid)
                    .Input(input_alpha)
                    .Attr("T", dtype)
                    .Finalize(g, &mul_2));

    TF_CHECK_OK(NodeBuilder(g->NewName("mul_3"), "Mul")
                    .Input(mul_2)
                    .Input(input_in)
                    .Attr("T", dtype)
                    .Finalize(g, &mul_3));

    TF_CHECK_OK(NodeBuilder(g->NewName("addv2_1"), "AddV2")
                    .Input(mul_1)
                    .Input(mul_3)
                    .Attr("T", dtype)
                    .Finalize(g, &addv2_1));
    return g;
  }

  // _MklDice
  TF_CHECK_OK(NodeBuilder(g->NewName("dice"), "_MklDice")
                         .Input(input_in)
                         .Input(input_rvar)
                         .Input(input_mean_rvar)
                         .Input(input_alpha)
                         .Attr("T", dtype)
                         .Finalize(g, &dice));
  return g;
}

#define BM_DICE(kind, ROWS, COLS, NTH)                                          \
  static void BM_DICE_##kind##_##ROWS##_##COLS##_##NTH##_CPU(                   \
      ::testing::benchmark::State& state) {                                     \
    state.SetItemsProcessed(static_cast<int64>(state.iterations())              \
                                              * ROWS * COLS * 3);               \
    SessionOptions opts;                                                        \
    opts.config.set_intra_op_parallelism_threads(NTH);                          \
    test::Benchmark("cpu", DiceGraph(#kind, ROWS, COLS), &opts).Run(state);     \
  }                                                                             \
  BENCHMARK(BM_DICE_##kind##_##ROWS##_##COLS##_##NTH##_CPU)->UseRealTime();

#define BENCHMARK_DICE(ROWS, COLS, NTH) \
  BM_DICE(Default, ROWS, COLS, NTH);    \
  BM_DICE(Mkl, ROWS, COLS, NTH);


#define BM_DICE_NTH(ROWS, COLS)         \
  BENCHMARK_DICE(ROWS, COLS, 1);        \
  BENCHMARK_DICE(ROWS, COLS, 4);        \
  BENCHMARK_DICE(ROWS, COLS, 8);

BM_DICE_NTH(40, 600);
BM_DICE_NTH(40, 400);
BM_DICE_NTH(40, 300);
BM_DICE_NTH(40, 200);
BM_DICE_NTH(40, 100);
BM_DICE_NTH(100, 600);
BM_DICE_NTH(100, 400);
BM_DICE_NTH(100, 300);
BM_DICE_NTH(100, 200);
BM_DICE_NTH(100, 100);
BM_DICE_NTH(200, 600);
BM_DICE_NTH(200, 400);
BM_DICE_NTH(200, 300);
BM_DICE_NTH(200, 200);
BM_DICE_NTH(200, 100);
BM_DICE_NTH(400, 600);
BM_DICE_NTH(400, 400);
BM_DICE_NTH(400, 300);
BM_DICE_NTH(400, 200);
BM_DICE_NTH(400, 100);
BM_DICE_NTH(500, 600);
BM_DICE_NTH(500, 400);
BM_DICE_NTH(500, 300);
BM_DICE_NTH(500, 200);
BM_DICE_NTH(500, 100);

}  // namespace
}  // namespace tensorflow

#endif  //INTEL_MKL

#endif  // AVX512F