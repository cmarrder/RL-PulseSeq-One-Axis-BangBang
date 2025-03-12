cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(collective)

set(CMAKE_PREFIX_PATH /usr/local/lib/libtorch)

find_package(Torch REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_RL_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(learn src/learn.cpp
	            include/Param.hpp
		    include/Sequence.hpp
                    include/Action.hpp
                    include/Crystal.hpp
                    include/Environment.hpp
                    include/Translate.hpp
                    include/Greedy.hpp
                    include/ReplayBuffer.hpp
                    include/Network.hpp
                    include/Agent.hpp
		    include/GaussQuad.hpp
                    )

target_include_directories(learn PRIVATE include)

target_link_libraries(learn "${TORCH_LIBRARIES}" Eigen3::Eigen)

set_property(TARGET learn PROPERTY CXX_STANDARD 17)
