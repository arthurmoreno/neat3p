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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <spdlog/spdlog.h>

#include <cstdint>

#include "config.hpp"
#include "genes.hpp"

// Create a shortcut for nanobind
namespace nb = nanobind;

NB_MODULE(_neat3p, m) {
    nb::class_<ConfigParameter>(m, "ConfigParameter")
        .def(nb::init<
                const std::string &,
                const nb::object &,
                std::optional<ConfigValue>
            >(),
            nb::arg("name"),
            nb::arg("value_type"),
            nb::arg("default") = std::nullopt
        )
        .def("repr", &ConfigParameter::repr)
        // Now parse() takes (section: object, config_parser: object).
        .def("parse", &ConfigParameter::parse,
            nb::arg("section"), nb::arg("config_parser"))
        .def("format", &ConfigParameter::format,
            nb::arg("value"))
        .def("interpret", &ConfigParameter::interpret,
            nb::arg("config_dict"))
        .def_rw("name", &ConfigParameter::name)
        .def_rw("value_type", &ConfigParameter::value_type)
        .def_rw("default", &ConfigParameter::default_value);


    nb::class_<GenomeConfig>(m, "GenomeConfig")
        .def(nb::init<>())
        .def_rw("compatibility_weight_coefficient", &GenomeConfig::compatibility_weight_coefficient);

    nb::class_<DefaultNodeGene>(m, "DefaultNodeGene")
        .def(nb::init<int>(), "Initialize with an integer key")
        .def_rw("key", &DefaultNodeGene::key)
        .def("copy", &DefaultNodeGene::copy)
        .def("distance", &DefaultNodeGene::distance);
}