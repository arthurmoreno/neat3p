"""Does general configuration parsing; used by other classes for their configuration."""

import os
import warnings
from configparser import ConfigParser

from ._neat3p import ConfigParameter


def write_pretty_params(f, config, params):
    param_names = [p.name for p in params]
    longest_name = max(len(name) for name in param_names)
    param_names.sort()
    params = dict((p.name, p) for p in params)

    for name in param_names:
        p = params[name]
        f.write(f"{p.name.ljust(longest_name)} = {p.format(getattr(config, p.name))}\n")


class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""

    pass


class DefaultClassConfig(object):
    """
    Replaces at least some boilerplate configuration code
    for reproduction, species_set, and stagnation classes.
    """

    def __init__(self, param_dict, param_list):
        self._params = param_list
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError(f"Unknown configuration item {unknown_list[0]!s}")

    @classmethod
    def write_config(cls, f, config):
        # pylint: disable=protected-access
        write_pretty_params(f, config, config._params)


class Config(object):
    """A container for user-configurable parameters of NEAT."""

    __params = [
        ConfigParameter("pop_size", int),
        ConfigParameter("fitness_criterion", str),
        ConfigParameter("fitness_threshold", float),
        ConfigParameter("reset_on_extinction", bool),
        ConfigParameter("no_fitness_termination", bool, False),
    ]

    def __init__(
        self,
        genome_type,
        reproduction_type,
        species_set_type,
        stagnation_type,
        filename,
        config_information=None,
    ):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, "parse_config")
        assert hasattr(reproduction_type, "parse_config")
        assert hasattr(species_set_type, "parse_config")
        assert hasattr(stagnation_type, "parse_config")

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.config_information = config_information

        if not os.path.isfile(filename):
            raise Exception("No such config file: " + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            parameters.read_file(f)

        # NEAT configuration
        if not parameters.has_section("NEAT"):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse("NEAT", parameters))
            else:
                try:
                    setattr(self, p.name, p.parse("NEAT", parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn(
                        f"Using default {p.default!r} for '{p.name!s}'",
                        DeprecationWarning,
                    )
            param_list_names.append(p.name)
        param_dict = dict(parameters.items("NEAT"))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError(
                    "Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(unknown_list)
                )
            raise UnknownConfigItemError(f"Unknown (section 'NEAT') configuration item {unknown_list[0]!s}")

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

    def save(self, filename):
        with open(filename, "w") as f:
            f.write("# The `NEAT` section specifies parameters particular to the NEAT algorithm\n")
            f.write("# or the experiment itself.  This is the only required section.\n")
            f.write("[NEAT]\n")
            write_pretty_params(f, self, self.__params)

            f.write(f"\n[{self.genome_type.__name__}]\n")
            self.genome_type.write_config(f, self.genome_config)

            f.write(f"\n[{self.species_set_type.__name__}]\n")
            self.species_set_type.write_config(f, self.species_set_config)

            f.write(f"\n[{self.stagnation_type.__name__}]\n")
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write(f"\n[{self.reproduction_type.__name__}]\n")
            self.reproduction_type.write_config(f, self.reproduction_config)
