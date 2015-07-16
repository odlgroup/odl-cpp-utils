#pragma once

#include <stdexcept>

template <typename R, typename T>
R narrow_cast(const T& t){
	R r = static_cast<R>(t);
	if (static_cast<T>(r) != t)
		throw std::invalid_argument("tried narrowing cast but information would be lost");
	return r;
}