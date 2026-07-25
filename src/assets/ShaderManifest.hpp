#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vkrt {
namespace assets {

struct ShaderEntry
{
    std::string name;
    std::string stage;
    std::string entry;
    std::string target;
    std::vector<std::string> deps;
};

class ShaderManifest
{
public:
    static ShaderManifest loadFromFile(const std::filesystem::path& manifestPath);

    const std::vector<ShaderEntry>& shaders() const { return m_shaders; }

private:
    std::vector<ShaderEntry> m_shaders;
};

} // namespace assets
} // namespace vkrt
