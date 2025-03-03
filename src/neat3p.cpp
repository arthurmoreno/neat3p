// #include <SDL2/SDL.h>
// #include <SDL2/SDL_image.h>
// #include <imgui/backends/imgui_impl_sdl2.h>
// #include <imgui/backends/imgui_impl_sdlrenderer2.h>
#define ENTT_ENTITY_TYPE int

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <spdlog/spdlog.h>

#include <cstdint>

#include "genes.hpp"

// Create a shortcut for nanobind
namespace nb = nanobind;

NB_MODULE(_neat3p, m) {

    // nb::class_<GenomeParams>(m, "GenomeParams")
    //     .def(nb::init<>())
    //     .def_rw("num_inputs", &GenomeParams::num_inputs)
    //     .def_rw("num_outputs", &GenomeParams::num_outputs)
    //     .def_rw("num_hidden", &GenomeParams::num_hidden)
    //     .def_rw("feed_forward", &GenomeParams::feed_forward)
    //     .def_rw("compatibility_disjoint_coefficient",
    //             &GenomeParams::compatibility_disjoint_coefficient)
    //     .def_rw("compatibility_weight_coefficient",
    //             &GenomeParams::compatibility_weight_coefficient)
    //     .def_rw("conn_add_prob", &GenomeParams::conn_add_prob)
    //     .def_rw("conn_delete_prob", &GenomeParams::conn_delete_prob)
    //     .def_rw("node_add_prob", &GenomeParams::node_add_prob)
    //     .def_rw("node_delete_prob", &GenomeParams::node_delete_prob)
    //     .def_rw("single_structural_mutation", 
    //             &GenomeParams::single_structural_mutation)
    //     .def_rw("structural_mutation_surer", 
    //             &GenomeParams::structural_mutation_surer)
    //     .def_rw("initial_connection",
    //             &GenomeParams::initial_connection);

    // nb::class_<DefaultGenomeConfig>(m, "DefaultGenomeConfig")
    //     .def(nb::init<const GenomeParams &>())
    //     .def_rw("num_inputs", &DefaultGenomeConfig::num_inputs)
    //     .def_rw("num_outputs", &DefaultGenomeConfig::num_outputs)
    //     .def_rw("num_hidden", &DefaultGenomeConfig::num_hidden)
    //     .def_rw("feed_forward", &DefaultGenomeConfig::feed_forward)
    //     .def_rw("compatibility_disjoint_coefficient",
    //         &DefaultGenomeConfig::compatibility_disjoint_coefficient)
    //     .def_rw("compatibility_weight_coefficient",
    //         &DefaultGenomeConfig::compatibility_weight_coefficient)
    //     .def_rw("conn_add_prob",
    //         &DefaultGenomeConfig::conn_add_prob)
    //     .def_rw("conn_delete_prob",
    //         &DefaultGenomeConfig::conn_delete_prob)
    //     .def_rw("node_add_prob",
    //         &DefaultGenomeConfig::node_add_prob) 
    //     .def_rw("node_delete_prob",
    //         &DefaultGenomeConfig::node_delete_prob)
    //     .def_rw("single_structural_mutation",
    //         &DefaultGenomeConfig::single_structural_mutation)
    //     .def_rw("structural_mutation_surer",
    //         &DefaultGenomeConfig::structural_mutation_surer)
    //     .def_rw("initial_connection",
    //         &DefaultGenomeConfig::initial_connection)
    //     .def_rw("connection_fraction",
    //         &DefaultGenomeConfig::connection_fraction)
    //     .def_rw("input_keys",
    //         &DefaultGenomeConfig::input_keys)
    //     .def_rw("output_keys",
    //         &DefaultGenomeConfig::output_keys)
    //     .def("get_new_node_key",
    //         &DefaultGenomeConfig::get_new_node_key);


    nb::class_<GenomeConfig>(m, "GenomeConfig")
        .def(nb::init<>())  // Default constructor
        .def_rw("compatibility_weight_coefficient", &GenomeConfig::compatibility_weight_coefficient);

    nb::class_<DefaultNodeGene>(m, "DefaultNodeGene")
        .def(nb::init<int>(), "Initialize with an integer key")
        .def_rw("key", &DefaultNodeGene::key)
        .def("copy", &DefaultNodeGene::copy)
        .def("distance", &DefaultNodeGene::distance);
        // .def("mutate", &DefaultNodeGene::mutate);

    // nb::class_<DefaultConnectionGene>(m, "DefaultConnectionGene")
    //     .def(nb::init<>())
    //     .def_rw("weight", &DefaultConnectionGene::weight)
    //     .def_rw("enabled", &DefaultConnectionGene::enabled)
    //     .def("copy", &DefaultConnectionGene::copy)
    //     .def("distance", &DefaultConnectionGene::distance);
    //     // .def("mutate", &DefaultConnectionGene::mutate)
    //     // .def("crossover", &DefaultConnectionGene::crossover);

    // nb::class_<DefaultGenome>(m, "DefaultGenome")
    //     .def(nb::init<>())
    //     .def_rw("key", &DefaultGenome::key)
    //     .def_rw("fitness", &DefaultGenome::fitness)
    //     .def_rw("nodes", &DefaultGenome::nodes)
    //     .def_rw("connections", &DefaultGenome::connections)
    //     .def("configure_new", &DefaultGenome::configure_new)
    //     .def("mutate_add_node", &DefaultGenome::mutate_add_node);

    // m.def("get_pruned_copy", &get_pruned_copy, "Get a pruned copy of the genome");
}