#pragma once

#include "RingBuffer.hpp"

#include <fmt/format.h>
#include <string>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <memory>

class Console {
public:
    Console() = default;

    static void log(const std::string& str) {
        instance->log0(str);
    }

    static void start() {
        instance->start0();
    }

    static void stop() {
        instance->stop0();
    }


private:
    void log0(const std::string& str) {
        _queue.push(str);
    }

    void start0() {
        using namespace std::chrono_literals;
        _running = true;
        _thread = std::move(std::thread{[this]{
            while(_running){
                while(!_queue.empty()) {
                    auto str = _queue.poll();
                    if (str.has_value()) {
//                        std::system("cls");
                        fmt::print("{}\n", *str);
//                        std::this_thread::sleep_for(100ms);
                    }
                }
            }
        }});
    }

    void stop0() {
        _running = false;
        _thread.join();
    }

private:
    RingBuffer<std::string, 256> _queue{};
    std::thread _thread;
    std::atomic_bool _running{};
    static std::unique_ptr<Console> instance;
};

std::unique_ptr<Console> Console::instance = std::make_unique<Console>();
