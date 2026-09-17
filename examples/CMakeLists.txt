add_executable(waltz waltz.cu)
target_link_libraries(waltz PRIVATE Waltz::waltz)
set_target_properties(
    waltz
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
               INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS waltz RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
