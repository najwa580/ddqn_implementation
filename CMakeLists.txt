cmake_minimum_required(VERSION 3.5)
project(my_bot)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================
# Dependencies
# =========================

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# =========================
# Install folders
# =========================

install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

# =========================
# Install Python package
# =========================

ament_python_install_package(my_bot)

install(PROGRAMS
  my_bot/train_q_learning.py
  DESTINATION lib/${PROJECT_NAME}
)

# =========================
# Testing
# =========================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

