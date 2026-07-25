#pragma once

#include "core/Math.hpp"


namespace vkrt {
namespace scene {
using namespace vkrt::core;

class Transform 
{
public:
    Transform();

    void setPosition(const glm::vec3& position);
    void translate(const glm::vec3& delta);
    glm::vec3 getPosition() const;

    void setRotation(const glm::vec3& eulerDegrees);
    void rotate(const glm::vec3& deltaDegrees);
    glm::vec3 getRotationEuler() const;
    glm::quat getRotationQuat() const;

    void setScale(const glm::vec3& scale);
    void scale(const glm::vec3& factor);
    glm::vec3 getScale() const;

    glm::mat4 getTransformMatrix() const;
    void setTransformMatrix(const glm::mat4& matrix);

    void printPosition() const;
    void printRotation() const;
    void printScale() const;

    glm::vec3 getForward() const;

private:
    glm::vec3 m_position;
    glm::quat m_rotation;
    glm::vec3 m_scale;
};

} // namespace scene
} // namespace vkrt
