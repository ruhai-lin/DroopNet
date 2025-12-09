# TinyGRU INT8 C Model Makefile
# ASIC Golden Reference Implementation

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Debug mode
DEBUG_FLAGS = -g -O0 -DDEBUG

# Source files
SRCS = main.c model.c inference.c
OBJS = $(SRCS:.c=.o)
TARGET = tiny_gru_test

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Run test
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

# Dependencies
main.o: main.c model.h inference.h
model.o: model.c model.h
inference.o: inference.c inference.h model.h

# Valgrind memory check
memcheck: debug
	valgrind --leak-check=full ./$(TARGET)

# Profiling
profile: CFLAGS += -pg
profile: clean $(TARGET)
	./$(TARGET)
	gprof $(TARGET) gmon.out > profile.txt
	@echo "Profile saved to profile.txt"

.PHONY: all clean run debug memcheck profile

