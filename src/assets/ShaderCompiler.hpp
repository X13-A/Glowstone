#pragma once

#include "assets/ShaderManifest.hpp"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>


namespace vkrt {
namespace assets {

class ShaderCompiler
{
public:
    explicit ShaderCompiler(ShaderManifest manifest);

    // Compiles every shader whose .spv is older than its .slang source or includes
    bool compileOutdated(const std::filesystem::path& shadersDir);

    // Sets a define that will be passed to all shaders during compilation
    // TODO: add a way to unset and clear them
    // TODO: or make it per-shader
    void setDefine(const std::string& name, const std::string& value);

    // Compile a single shader without checking timestamps
    bool compileShader(const std::string& name, const std::filesystem::path& shadersDir) const;

private:
    // Empty if slangc is not installed
    static std::filesystem::path findSlangc();
    
    static std::filesystem::file_time_type writeTime(const std::filesystem::path& path);

    const ShaderEntry* findShader(const std::string& name) const;
    std::string defineArguments() const;
    bool isOutdated(const ShaderEntry& shader, const std::filesystem::path& shadersDir) const;
    std::vector<const ShaderEntry*> collectOutdated(const std::filesystem::path& shadersDir) const;
    bool compile(const ShaderEntry& shader, const std::filesystem::path& shadersDir) const;
    void validateSPIRV(const std::filesystem::path& shadersDir) const;

    ShaderManifest m_manifest;
    std::filesystem::path m_slangc;
    std::vector<std::pair<std::string, std::string>> m_defines;
};

} // namespace assets
} // namespace vkrt
