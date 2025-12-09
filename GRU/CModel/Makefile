# TinyGRU INT8 C Model Makefile
# ASIC Golden Reference Implementation

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Debug 模式
DEBUG_FLAGS = -g -O0 -DDEBUG

# 源文件
SRCS = main.c model.c inference.c
OBJS = $(SRCS:.c=.o)
TARGET = tiny_gru_test

# 默认目标
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug 版本
debug: CFLAGS = $(DEBUG_FLAGS)
debug: clean $(TARGET)

# 运行测试
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 依赖关系
main.o: main.c model.h inference.h
model.o: model.c model.h
inference.o: inference.c inference.h model.h

# Valgrind 内存检查
memcheck: debug
	valgrind --leak-check=full ./$(TARGET)

# 性能分析
profile: CFLAGS += -pg
profile: clean $(TARGET)
	./$(TARGET)
	gprof $(TARGET) gmon.out > profile.txt
	@echo "Profile saved to profile.txt"

.PHONY: all clean run debug memcheck profile

