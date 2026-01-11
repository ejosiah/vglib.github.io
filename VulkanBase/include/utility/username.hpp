#pragma once

#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
    #include <sys/types.h>
    #include <pwd.h>
#endif

inline std::string getOSUsername()
{
#ifdef _WIN32
    char username[256];
    DWORD size = sizeof(username);

    if (GetUserNameA(username, &size)) {
        return std::string(username);
    }

    return std::string{};
#else
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_name) {
        return std::string(pw->pw_name);
    }

    return std::string{};
#endif
}
