#pragma once

#ifdef WIN32
#include <Windows.h>
#undef near
#undef far
#undef MAX_FLOAT
#undef MIN_FLOAT
#undef max
#undef min
#undef DELETE
#undef APIENTRY
#endif