NAME = NeuralNet_CPP
CC = g++
FLAGS = -Wall -Werror -Wextra -std=c++17 \
	-I./NeuralNetwork/include/Eigen \
	-I./NeuralNetwork/include \
	-I./mnist/include

# source files
NNCDIR = ./NeuralNetwork/srcs
SRCS = 	./main.cpp	\
		$(NNCDIR)/utils.cpp	\
		$(NNCDIR)/Network.cpp

OBJDIR = ./build
OBJS = $(SRCS:%.c=$(OBJDIR)/%.o)

# compile step
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(FLAGS) -c $< -o $@

# link step
$(NAME): $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $(NAME)

# create build dir
$(OBJDIR):
	mkdir -p $(OBJDIR)

all: $(OBJDIR) $(OBJS) $(NAME)

clean:
	rm -fr $(OBJDIR)

fclean: clean
	rm -fr $(NAME)

re: fclean all

.PHONY: all clean fclean re
