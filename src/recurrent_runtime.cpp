// Torch-free evaluator of a recurrent NEAT phenotype (see neat3p/runtime/layout.py for the layout).
//
// Built WITHOUT fast-math (CMakeLists.txt): NaN/inf must propagate and sums keep IEEE semantics so
// the results match torch's RecurrentNet.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

template <typename Scalar>
class RecurrentKernel {
   public:
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    RecurrentKernel(const Eigen::MatrixXd &input_to_hidden, const Eigen::MatrixXd &hidden_to_hidden,
                    const Eigen::MatrixXd &output_to_hidden, const Eigen::MatrixXd &input_to_output,
                    const Eigen::MatrixXd &hidden_to_output,
                    const Eigen::MatrixXd &output_to_output, const Eigen::VectorXd &hidden_biases,
                    const Eigen::VectorXd &hidden_responses, const Eigen::VectorXd &output_biases,
                    const Eigen::VectorXd &output_responses, bool use_current_activs,
                    int n_internal_steps)
        : n_inputs_(input_to_output.cols()),
          n_hidden_(hidden_biases.size()),
          n_outputs_(output_biases.size()),
          use_current_activs_(use_current_activs),
          n_internal_steps_(n_internal_steps),
          w_ih_(input_to_hidden.cast<Scalar>()),
          w_hh_(hidden_to_hidden.cast<Scalar>()),
          w_oh_(output_to_hidden.cast<Scalar>()),
          w_io_(input_to_output.cast<Scalar>()),
          w_ho_(hidden_to_output.cast<Scalar>()),
          w_oo_(output_to_output.cast<Scalar>()),
          b_h_(hidden_biases.cast<Scalar>()),
          r_h_(hidden_responses.cast<Scalar>()),
          b_o_(output_biases.cast<Scalar>()),
          r_o_(output_responses.cast<Scalar>()) {
        if (n_internal_steps < 1) throw std::invalid_argument("n_internal_steps must be >= 1");
        check(hidden_responses.size() == n_hidden_, "hidden_responses length");
        check(output_responses.size() == n_outputs_, "output_responses length");
        check(w_ih_.rows() == n_hidden_ && w_ih_.cols() == n_inputs_, "input_to_hidden shape");
        check(w_hh_.rows() == n_hidden_ && w_hh_.cols() == n_hidden_, "hidden_to_hidden shape");
        check(w_oh_.rows() == n_hidden_ && w_oh_.cols() == n_outputs_, "output_to_hidden shape");
        check(w_io_.rows() == n_outputs_ && w_io_.cols() == n_inputs_, "input_to_output shape");
        check(w_ho_.rows() == n_outputs_ && w_ho_.cols() == n_hidden_, "hidden_to_output shape");
        check(w_oo_.rows() == n_outputs_ && w_oo_.cols() == n_outputs_, "output_to_output shape");
        reset();
    }

    void reset() {
        h_ = Vec::Zero(n_hidden_);
        o_ = Vec::Zero(n_outputs_);
    }

    // x: n_inputs values; writes n_outputs values into out.
    void activate(const double *x_in, double *out) {
        Vec x = Eigen::Map<const Eigen::VectorXd>(x_in, n_inputs_).template cast<Scalar>();
        Vec h_for_output = h_;
        if (n_hidden_ > 0) {
            for (int s = 0; s < n_internal_steps_; ++s) {
                Vec z = w_ih_ * x;
                z += w_hh_ * h_;
                z += w_oh_ * o_;
                h_ = sigmoid5(z.cwiseProduct(r_h_) + b_h_);
            }
            if (use_current_activs_) h_for_output = h_;
        }
        Vec z = w_io_ * x;
        z += w_oo_ * o_;
        if (n_hidden_ > 0) z += w_ho_ * h_for_output;
        o_ = sigmoid5(z.cwiseProduct(r_o_) + b_o_);
        for (Eigen::Index i = 0; i < n_outputs_; ++i) out[i] = static_cast<double>(o_[i]);
    }

    Eigen::Index n_inputs() const { return n_inputs_; }
    Eigen::Index n_hidden() const { return n_hidden_; }
    Eigen::Index n_outputs() const { return n_outputs_; }
    Eigen::VectorXd hidden_state() const { return h_.template cast<double>(); }
    Eigen::VectorXd output_state() const { return o_.template cast<double>(); }

   private:
    static void check(bool ok, const char *what) {
        if (!ok) throw std::invalid_argument(std::string("recurrent kernel: bad ") + what);
    }

    // sigmoid(5 z) with scalar std::exp: exact IEEE behaviour for NaN, inf and overflow.
    static Vec sigmoid5(const Vec &z) {
        Vec r(z.size());
        for (Eigen::Index i = 0; i < z.size(); ++i)
            r[i] = Scalar(1) / (Scalar(1) + std::exp(Scalar(-5) * z[i]));
        return r;
    }

    Eigen::Index n_inputs_, n_hidden_, n_outputs_;
    bool use_current_activs_;
    int n_internal_steps_;
    Mat w_ih_, w_hh_, w_oh_, w_io_, w_ho_, w_oo_;
    Vec b_h_, r_h_, b_o_, r_o_;
    Vec h_, o_;
};

template <typename Scalar>
void bind_kernel(nb::module_ &m, const char *name) {
    using K = RecurrentKernel<Scalar>;
    using Out = nb::ndarray<nb::numpy, double, nb::shape<-1>>;
    using In = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

    nb::class_<K>(m, name)
        .def(nb::init<const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                      const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                      const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &,
                      const Eigen::VectorXd &, bool, int>(),
             nb::arg("input_to_hidden"), nb::arg("hidden_to_hidden"), nb::arg("output_to_hidden"),
             nb::arg("input_to_output"), nb::arg("hidden_to_output"), nb::arg("output_to_output"),
             nb::arg("hidden_biases"), nb::arg("hidden_responses"), nb::arg("output_biases"),
             nb::arg("output_responses"), nb::arg("use_current_activs"),
             nb::arg("n_internal_steps") = 1)
        .def(
            "activate",
            [](K &k, In x) -> Out {
                if (static_cast<Eigen::Index>(x.shape(0)) != k.n_inputs())
                    throw std::invalid_argument("expected input of length " +
                                                std::to_string(k.n_inputs()) + ", got " +
                                                std::to_string(x.shape(0)));
                const size_t n = static_cast<size_t>(k.n_outputs());
                double *out = new double[n];
                nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });
                {
                    nb::gil_scoped_release release;
                    k.activate(x.data(), out);
                }
                return Out(out, {n}, owner);
            },
            nb::arg("x"))
        .def("reset", &K::reset)
        .def_prop_ro("n_inputs", &K::n_inputs)
        .def_prop_ro("n_hidden", &K::n_hidden)
        .def_prop_ro("n_outputs", &K::n_outputs)
        .def_prop_ro("hidden_state", &K::hidden_state)
        .def_prop_ro("output_state", &K::output_state);
}

}  // namespace

void bind_recurrent_runtime(nb::module_ &m) {
    bind_kernel<double>(m, "RecurrentKernelF64");
    bind_kernel<float>(m, "RecurrentKernelF32");
}
