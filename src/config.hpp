#ifndef NEAT_CONFIG_HPP
#define NEAT_CONFIG_HPP

// Include nanobind headers because we need nb::object in the constructor.
#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

// A variant type to hold a configuration value.
using ConfigValue = std::variant<int, bool, float, std::string, std::vector<std::string>, nb::object>;

class ConfigParameter {
public:
    std::string name;
    // This member holds a string (e.g. "int", "bool", "float", "list", "str") extracted from the Python type's __name__.
    std::string value_type;
    std::optional<ConfigValue> default_value;

    // Constructor accepts a Python type object for the value type.
    ConfigParameter(const std::string &name, const nb::object &py_type, 
                    std::optional<ConfigValue> default_value = std::nullopt)
        : name(name),
          // Convert the Python type's __name__ to a std::string
          value_type(nb::cast<std::string>(nb::getattr(py_type, "__name__"))),
          default_value(default_value)
    {}

    // Mimics Python's __repr__.
    std::string repr() const {
        std::ostringstream oss;
        oss << "ConfigParameter(" << quote(name) << ", " << quote(value_type);
        if (!default_value.has_value()) {
            oss << ")";
        } else {
            oss << ", " << configValueToString(*default_value) << ")";
        }
        return oss.str();
    }

    /**
     * parse(section, config_parser)
     *
     * This version of parse() expects:
     *   - section: a Python string (the INI section name)
     *   - config_parser: a Python configparser.ConfigParser object
     *
     * It replicates the Python logic:
     *
     *   if self.value_type == int:
     *       return config_parser.getint(section, self.name)
     *   ...
     */
    ConfigValue parse(nb::object section, nb::object config_parser) const {
        // Convert the 'section' object to a std::string
        std::string section_str = nb::cast<std::string>(section);

        try {
            if (value_type == "int") {
                nb::object result = config_parser.attr("getint")(section_str, name);
                return nb::cast<int>(result);
            } 
            else if (value_type == "bool") {
                nb::object result = config_parser.attr("getboolean")(section_str, name);
                return nb::cast<bool>(result);
            }
            else if (value_type == "float") {
                nb::object result = config_parser.attr("getfloat")(section_str, name);
                return nb::cast<float>(result);
            }
            else if (value_type == "list") {
                // config_parser.get(...) returns a string -> we split it
                nb::object result = config_parser.attr("get")(section_str, name);
                std::string s = nb::cast<std::string>(result);
                return split(s);
            }
            else if (value_type == "str") {
                nb::object result = config_parser.attr("get")(section_str, name);
                return nb::cast<std::string>(result);
            }
            else {
                throw std::runtime_error("Unexpected configuration type: " + value_type);
            }
        } catch (const std::exception &e) {
            throw std::runtime_error("Error parsing config item '" + name + "': " + e.what());
        }
    }

    // Interprets a configuration value from a nb::dict.
    ConfigValue interpret(nb::dict config_dict) const {
        // Check if the dictionary contains our key.
        if (!config_dict.contains(name.c_str())) {
            if (!default_value.has_value()) {
                throw std::runtime_error("Missing configuration item: " + name);
            } else {
                std::cerr << "Warning: Using default " << configValueToString(*default_value)
                        << " for '" << name << "'\n";
                return *default_value;
            }
        }
        // Retrieve the value as a string from the nb::dict.
        nb::object value_obj = config_dict[name.c_str()];
        std::string value_str = nb::cast<std::string>(value_obj);

        try {
            if (value_type == "str")
                return value_str;
            else if (value_type == "int")
                return std::stoi(value_str);
            else if (value_type == "bool") {
                std::string lower = toLower(value_str);
                if (lower == "true")
                    return nb::bool_(true);
                else if (lower == "false") {
                    std::cout << "HERE JUST BEFORE FALSE!" << std::endl;
                    return nb::bool_(false);
                }
                else
                    throw std::runtime_error(name + " must be True or False");
            }
            else if (value_type == "float")
                return std::stof(value_str);
            else if (value_type == "list")
                return split(value_str);
            else
                throw std::runtime_error("Unexpected configuration type: " + value_type);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error interpreting config item '" + name +
                                    "' with value '" + value_str + "': " + e.what());
        }
    }

    // Formats a given configuration value to a string (unchanged).
    std::string format(const ConfigValue &value) const {
        if (value_type == "list") {
            const auto &vec = std::get<std::vector<std::string>>(value);
            std::ostringstream oss;
            for (size_t i = 0; i < vec.size(); ++i) {
                oss << vec[i];
                if (i != vec.size() - 1)
                    oss << " ";
            }
            return oss.str();
        } else {
            return configValueToString(value);
        }
    }

private:
    // Helper: converts a ConfigValue variant to a string.
    static std::string configValueToString(const ConfigValue &val) {
        return std::visit([](auto &&arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::is_same_v<T, int>) {
                return std::to_string(arg);
            } else if constexpr(std::is_same_v<T, bool>) {
                return arg ? "True" : "False";
            } else if constexpr(std::is_same_v<T, float>) {
                return std::to_string(arg);
            } else if constexpr(std::is_same_v<T, std::string>) {
                return arg;
            } else if constexpr(std::is_same_v<T, std::vector<std::string>>) {
                std::ostringstream oss;
                for (size_t i = 0; i < arg.size(); ++i) {
                    oss << arg[i];
                    if (i != arg.size() - 1)
                        oss << " ";
                }
                return oss.str();
            } else {
                return "";
            }
        }, val);
    }

    // Helper: returns a lowercase copy of a string.
    static std::string toLower(const std::string &s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower;
    }

    // Helper: splits a string by whitespace.
    static std::vector<std::string> split(const std::string &s) {
        std::istringstream iss(s);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token)
            tokens.push_back(token);
        return tokens;
    }

    // Helper: wraps a string in quotes.
    static std::string quote(const std::string &s) {
        return "\"" + s + "\"";
    }
};

#endif // NEAT_CONFIG_HPP