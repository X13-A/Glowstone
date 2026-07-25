#include "assets/ShaderManifest.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>


namespace vkrt {
namespace assets {

namespace {

std::string requireString(const nlohmann::json& object,
                          const char* key,
                          const std::string& fallback,
                          const std::string& context)
{
    auto it = object.find(key);
    if (it == object.end())
    {
        if (!fallback.empty())
        {
            return fallback;
        }
        throw std::runtime_error(context + ": missing required string member '" + key + "'");
    }

    if (!it->is_string())
    {
        throw std::runtime_error(context + ": member '" + key + "' must be a string");
    }
    return it->get<std::string>();
}

} // namespace

ShaderManifest ShaderManifest::loadFromFile(const std::filesystem::path& manifestPath)
{
    std::ifstream file(manifestPath);
    if (!file)
    {
        throw std::runtime_error("Shader manifest not found: " + manifestPath.string());
    }

    nlohmann::json root;
    try
    {
        file >> root;
    }
    catch (const nlohmann::json::parse_error& e)
    {
        throw std::runtime_error("Shader manifest " + manifestPath.string() + " is malformed: " + e.what());
    }

    std::string defaultTarget;
    std::string defaultEntry;
    if (auto defaults = root.find("defaults"); defaults != root.end())
    {
        defaultTarget = defaults->value("target", std::string{});
        defaultEntry  = defaults->value("entry", std::string{});
    }

    ShaderManifest manifest;

    auto shaders = root.find("shaders");
    if (shaders == root.end() || !shaders->is_array())
    {
        throw std::runtime_error("Shader manifest " + manifestPath.string() + ": missing 'shaders' array");
    }

    manifest.m_shaders.reserve(shaders->size());
    for (const auto& node : *shaders)
    {
        const std::string context = "Shader manifest " + manifestPath.string();

        ShaderEntry shader;
        shader.name   = requireString(node, "name", {}, context);
        shader.stage  = requireString(node, "stage", {}, context + " (" + shader.name + ")");
        shader.entry  = requireString(node, "entry", defaultEntry, context + " (" + shader.name + ")");
        shader.target = requireString(node, "target", defaultTarget, context + " (" + shader.name + ")");

        if (auto deps = node.find("deps"); deps != node.end())
        {
            for (const auto& dep : *deps)
            {
                shader.deps.push_back(dep.get<std::string>());
            }
        }

        manifest.m_shaders.push_back(std::move(shader));
    }

    return manifest;
}

} // namespace assets
} // namespace vkrt
