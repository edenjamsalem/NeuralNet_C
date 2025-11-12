NAME = NN_Mnist
CC = g++
FLAGS = -Wall -Werror -Wextra -std=c++17 \
	-I./NeuralNetwork/include/Eigen \
	-I./NeuralNetwork/include \
	-I./mnist/include

# source files
NNDIR = ./NeuralNetwork/source
SRCS = 	./main.cpp	\
		$(NNDIR)/utils.cpp	\
		$(NNDIR)/Network.cpp

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
