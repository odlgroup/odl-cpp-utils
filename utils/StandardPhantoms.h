#pragma once

#include <utils/Ellipse.h>
#include <vector>

#include <Eigen/Dense>

namespace SimRec2D {
/**
 * Enum of all phantom types supported by getPhantomParameters
 */
enum class PhantomType {
    sheppLogan,         /// the standard Shepp-Logan phantom
    modifiedSheppLogan, /// Shepp-Logan with improved contrast
    twoEllipses,        /// A very simple phantom with two structures
};

/**
 * Gets the ellipse parameters for the phantom given by @a type
 */
std::vector<Ellipse> getPhantomParameters(PhantomType type);
}