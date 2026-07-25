#include "assets/ShaderCompiler.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>


namespace vkrt {
namespace assets {

namespace fs = std::filesystem;

namespace {

std::string shellCommand(const std::string& program, const std::string& arguments)
{
    std::string command = "\"" + program + "\" " + arguments;
#ifdef _WIN32
    command = "\"" + command + "\"";
#endif
    return command;
}

std::string quote(const fs::path& path)
{
    return "\"" + path.string() + "\"";
}

} // namespace

ShaderCompiler::ShaderCompiler(ShaderManifest manifest)
    : m_manifest(std::move(manifest))
    , m_slangc(findSlangc())
{
}

fs::path ShaderCompiler::findSlangc()
{
#ifdef _WIN32
    const char* executable = "slangc.exe";
    const char separator = ';';
#else
    const char* executable = "slangc";
    const char separator = ':';
#endif

    std::vector<fs::path> directories;
    if (const char* sdk = std::getenv("VULKAN_SDK"))
    {
        directories.emplace_back(fs::path(sdk) / "Bin");
        directories.emplace_back(fs::path(sdk) / "bin");
    }

    if (const char* path = std::getenv("PATH"))
    {
        std::istringstream stream(path);
        std::string directory;
        while (std::getline(stream, directory, separator))
        {
            if (!directory.empty())
            {
                directories.emplace_back(directory);
            }
        }
    }

    for (const auto& directory : directories)
    {
        std::error_code ec;
        const fs::path candidate = directory / executable;
        if (fs::exists(candidate, ec))
        {
            return candidate;
        }
    }

    return {};
}

fs::file_time_type ShaderCompiler::writeTime(const fs::path& path)
{
    std::error_code ec;
    const auto time = fs::last_write_time(path, ec);
    return ec ? fs::file_time_type{} : time;
}

bool ShaderCompiler::isOutdated(const ShaderEntry& shader, const fs::path& shadersDir) const
{
    const fs::path source = shadersDir / (shader.name + ".slang");
    if (!fs::exists(source))
    {
        // No Slang source file present
        return false;
    }

    const auto spvTime = writeTime(shadersDir / (shader.name + ".spv"));
    if (spvTime == fs::file_time_type{})
    {
        return true;
    }

    if (writeTime(source) > spvTime)
    {
        return true;
    }

    for (const auto& dep : shader.deps)
    {
        if (writeTime(shadersDir / (dep + ".slang")) > spvTime)
        {
            return true;
        }
    }

    return false;
}

std::vector<const ShaderEntry*> ShaderCompiler::collectOutdated(const fs::path& shadersDir) const
{
    std::vector<const ShaderEntry*> outdated;
    for (const auto& shader : m_manifest.shaders())
    {
        if (isOutdated(shader, shadersDir))
        {
            outdated.push_back(&shader);
        }
    }
    return outdated;
}

std::string ShaderCompiler::defineArguments() const
{
    std::string arguments;
    for (const auto& define : m_defines)
    {
        arguments += " -D" + define.first + "=" + define.second;
    }
    return arguments;
}

bool ShaderCompiler::compile(const ShaderEntry& shader, const fs::path& shadersDir) const
{
    const fs::path source = shadersDir / (shader.name + ".slang");
    const fs::path output = shadersDir / (shader.name + ".spv");

    const std::string arguments =
        "-target " + shader.target +
        " -stage " + shader.stage +
        " -entry " + shader.entry +
        defineArguments() +
        " -o " + quote(output) +
        " " + quote(source);

    return std::system(shellCommand(m_slangc.string(), arguments).c_str()) == 0;
}

void ShaderCompiler::setDefine(const std::string& name, const std::string& value)
{
    for (auto& define : m_defines)
    {
        if (define.first == name)
        {
            define.second = value;
            return;
        }
    }
    m_defines.emplace_back(name, value);
}

const ShaderEntry* ShaderCompiler::findShader(const std::string& name) const
{
    for (const auto& shader : m_manifest.shaders())
    {
        if (shader.name == name)
        {
            return &shader;
        }
    }
    return nullptr;
}

bool ShaderCompiler::compileShader(const std::string& name, const fs::path& shadersDir) const
{
    if (m_slangc.empty())
    {
        std::cerr << "slangc was not found; cannot compile " << name << std::endl;
        return false;
    }

    const ShaderEntry* shader = findShader(name);
    if (shader == nullptr)
    {
        std::cerr << "Shader '" << name << "' is not in the manifest." << std::endl;
        return false;
    }

    if (!compile(*shader, shadersDir))
    {
        std::cerr << "Failed to compile " << name << ".slang." << std::endl;
        return false;
    }

    return true;
}

void ShaderCompiler::validateSPIRV(const fs::path& shadersDir) const
{
    std::vector<std::string> missing;
    for (const auto& shader : m_manifest.shaders())
    {
        if (!fs::exists(shadersDir / (shader.name + ".spv")))
        {
            missing.push_back(shader.name + ".spv");
        }
    }

    if (missing.empty())
    {
        return;
    }

    std::ostringstream message;
    message << "Missing SPIR-V in " << shadersDir.string() << ":";
    for (const auto& name : missing)
    {
        message << "\n  " << name;
    }
    message << (m_slangc.empty()
        ? "\nslangc was not found, so these could not be compiled."
        : "\nCompilation failed.");

    throw std::runtime_error(message.str());
}

bool ShaderCompiler::compileOutdated(const fs::path& shadersDir)
{
    if (!fs::exists(shadersDir))
    {
        throw std::runtime_error("Shaders directory does not exist: " + shadersDir.string());
    }

    bool compiled = false;
    if (!m_slangc.empty())
    {
        for (const ShaderEntry* shader : collectOutdated(shadersDir))
        {
            std::cout << "Compiling: " << shader->name << ".slang" << std::endl;
            if (compile(*shader, shadersDir))
            {
                compiled = true;
            }
            else
            {
                std::cerr << "Failed to compile " << shader->name << ".slang." << std::endl;
            }
        }
    }

    // Fatal if it leaves any shader without SPIR-V
    validateSPIRV(shadersDir);
    return compiled;
}

} // namespace assets
} // namespace vkrt
