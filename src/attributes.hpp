#ifndef ATTRIBUTES_HPP
#define ATTRIBUTES_HPP

#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "config.hpp"  // added to access ConfigParameter
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace neat3p {

// A simple config struct holding attribute parameters.
struct AttributeConfig {
    // FloatAttribute parameters
    double init_mean;
    double init_stdev;
    std::string init_type;
    double replace_rate_f;
    double mutate_rate_f;
    double mutate_power_f;
    double max_value_f;
    double min_value_f;
    // IntegerAttribute parameters
    double mutate_rate_i;
    double mutate_power_i;
    int replace_rate_i; // use int for simplicity
    int max_value_i;
    int min_value_i;
    // BoolAttribute parameters
    std::string default_bool;
    double mutate_rate_b;
    double rate_to_true_add;
    double rate_to_false_add;
    // StringAttribute parameters
    std::string default_str;
    std::vector<std::string> options;
    double mutate_rate_s;
};


class BaseAttribute {
public:
    std::string name;
protected:
    // New: store config item type and default value as strings.
    std::unordered_map<std::string, std::pair<std::string, std::string>> _config_items;
    // New: store generated config item names.
    std::unordered_map<std::string, std::string> _config_item_names;
public:
    // New: Helper to convert nb::dict to std::unordered_map
    static std::unordered_map<std::string, std::string> convert_dict(const nb::dict &d) {
        std::unordered_map<std::string, std::string> res;
        for (auto item : d) {
            std::string key = nb::cast<std::string>(item.first);
            std::string value = nb::cast<std::string>(item.second);
            res[key] = value;
        }
        return res;
    }

    // Modified constructor accepting a std::unordered_map
    BaseAttribute(const std::string &name, const std::unordered_map<std::string, std::string> &default_dict = {})
        : name(name)
    {
        // Assume _config_items is populated by derived classes.
        for (const auto &d : default_dict) {
            auto it = _config_items.find(d.first);
            if (it != _config_items.end()) {
                it->second.second = d.second;
            }
        }
        // Generate config item names.
        for (const auto &kv : _config_items) {
            _config_item_names[kv.first] = config_item_name(kv.first);
        }
    }
    
    // New: Overloaded constructor receiving nb::dict
    BaseAttribute(const std::string &name, const nb::dict &default_dict)
        : BaseAttribute(name, convert_dict(default_dict)) {}

    virtual ~BaseAttribute() = default;

    std::string config_item_name(const std::string &config_item_base_name) const {
        return name + "_" + config_item_base_name;
    }

    // NEW: Provide a get_config_params method
    virtual std::vector<ConfigParameter> get_config_params() const {
        return {};  // return empty vector by default
    }
};

class FloatAttribute : public BaseAttribute {
public:
    // Existing constructor.
    FloatAttribute(const std::string &name) : BaseAttribute(name) {}
    // NEW: Additional constructor overload with nb::dict.
    FloatAttribute(const std::string &name, const nb::dict &default_dict)
        : BaseAttribute(name, default_dict) {}

    double clamp(double value, const AttributeConfig &config) const {
        return std::max(std::min(value, config.max_value_f), config.min_value_f);
    }
    
    double init_value(const AttributeConfig &config) const {
        // Use a simple random engine
        static std::random_device rd;
        static std::mt19937 gen(rd());
        if (config.init_type.find("gauss") != std::string::npos ||
            config.init_type.find("normal") != std::string::npos) {
            std::normal_distribution<> d(config.init_mean, config.init_stdev);
            return clamp(d(gen), config);
        }
        if (config.init_type.find("uniform") != std::string::npos) {
            double low = std::max(config.min_value_f, config.init_mean - 2*config.init_stdev);
            double high = std::min(config.max_value_f, config.init_mean + 2*config.init_stdev);
            std::uniform_real_distribution<> d(low, high);
            return d(gen);
        }
        throw std::runtime_error("Unknown init_type for " + name);
    }
    
    double mutate_value(double value, const AttributeConfig &config) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        double r = dist(gen);
        if (r < config.mutate_rate_f) {
            std::normal_distribution<> d(0.0, config.mutate_power_f);
            return clamp(value + d(gen), config);
        }
        if (r < (config.replace_rate_f + config.mutate_rate_f)) {
            return init_value(config);
        }
        return value;
    }
    
    void validate(const AttributeConfig &config) const {
        if (config.max_value_f < config.min_value_f)
            throw std::runtime_error("Invalid min/max configuration for " + name);
    }
    
    // NEW: Implement get_config_params (customize as needed)
    std::vector<ConfigParameter> get_config_params() const override {
        // For now, return an empty vector.
        return {};
    }
};

class IntegerAttribute : public BaseAttribute {
public:
    IntegerAttribute(const std::string &name) : BaseAttribute(name) {}
    IntegerAttribute(const std::string &name, const nb::dict &default_dict)
        : BaseAttribute(name, default_dict) {}

    int clamp(int value, const AttributeConfig &config) const {
        return std::max(std::min(value, config.max_value_i), config.min_value_i);
    }
    
    int init_value(const AttributeConfig &config) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> d(config.min_value_i, config.max_value_i);
        return d(gen);
    }
    
    int mutate_value(int value, const AttributeConfig &config) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        double r = dist(gen);
        if (r < config.mutate_rate_i) {
            std::normal_distribution<> d(0.0, config.mutate_power_i);
            return clamp(value + static_cast<int>(std::round(d(gen))), config);
        }
        if (r < (config.replace_rate_i + config.mutate_rate_i)) {
            return init_value(config);
        }
        return value;
    }
    
    void validate(const AttributeConfig &config) const {
        if (config.max_value_i < config.min_value_i)
            throw std::runtime_error("Invalid min/max configuration for " + name);
    }
    
    // NEW: Implement get_config_params
    std::vector<ConfigParameter> get_config_params() const override {
        return {};
    }
};

class BoolAttribute : public BaseAttribute {
public:
    BoolAttribute(const std::string &name) : BaseAttribute(name) {}
    BoolAttribute(const std::string &name, const nb::dict &default_dict)
        : BaseAttribute(name, default_dict) {}

    bool init_value(const AttributeConfig &config) const {
        std::string def = config.default_bool;
        std::transform(def.begin(), def.end(), def.begin(), ::tolower);
        if (def == "1" || def == "on" || def == "yes" || def == "true")
            return true;
        if (def == "0" || def == "off" || def == "no" || def == "false")
            return false;
        if (def == "random" || def == "none") {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(0.0, 1.0);
            return dist(gen) < 0.5;
        }
        throw std::runtime_error("Unknown default value for " + name);
    }
    
    bool mutate_value(bool value, const AttributeConfig &config) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        double r = std::uniform_real_distribution<>(0.0, 1.0)(gen);
        double rate = config.mutate_rate_b;
        rate += (value ? config.rate_to_false_add : config.rate_to_true_add);
        if (r < rate) {
            return std::uniform_real_distribution<>(0.0, 1.0)(gen) < 0.5;
        }
        return value;
    }
    
    void validate(const AttributeConfig &config) const {
        std::string def = config.default_bool;
        std::transform(def.begin(), def.end(), def.begin(), ::tolower);
        if (!(def=="1" || def=="on" || def=="yes" || def=="true" ||
              def=="0" || def=="off" || def=="no" || def=="false" ||
              def=="random" || def=="none"))
            throw std::runtime_error("Invalid default value for " + name);
    }
    
    // NEW: Implement get_config_params
    std::vector<ConfigParameter> get_config_params() const override {
        return {};
    }
};

class StringAttribute : public BaseAttribute {
public:
    StringAttribute(const std::string &name) : BaseAttribute(name) {}
    StringAttribute(const std::string &name, const nb::dict &default_dict)
        : BaseAttribute(name, default_dict) {}

    std::string init_value(const AttributeConfig &config) const {
        std::string def = config.default_str;
        std::string low = def;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        if (low == "none" || low == "random") {
            if (config.options.empty())
                throw std::runtime_error("No options provided for " + name);
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> d(0, config.options.size()-1);
            return config.options[d(gen)];
        }
        return def;
    }
    
    std::string mutate_value(const std::string &value, const AttributeConfig &config) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        if (config.mutate_rate_s > 0 &&
            std::uniform_real_distribution<>(0.0, 1.0)(gen) < config.mutate_rate_s) {
            if (config.options.empty())
                throw std::runtime_error("No options provided for " + name);
            std::uniform_int_distribution<> d(0, config.options.size()-1);
            return config.options[d(gen)];
        }
        return value;
    }
    
    void validate(const AttributeConfig &config) const {
        std::string def = config.default_str;
        if (def != "none" && def != "random") {
            auto it = std::find(config.options.begin(), config.options.end(), def);
            if (it == config.options.end())
                throw std::runtime_error("Invalid initial value " + def + " for " + name);
        }
    }
    
    // NEW: Implement get_config_params
    std::vector<ConfigParameter> get_config_params() const override {
        return {};
    }
};

} // namespace neat3p

#endif // ATTRIBUTES_HPP
