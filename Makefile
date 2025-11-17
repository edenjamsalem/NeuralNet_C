NAME = NN_Mnist
CC = g++
FLAGS = -O3 -DNDEBUG -march=native -std=c++17 \
	-Wall -Werror -Wextra \
	-I./NeuralNetwork/include/Eigen \
	-I./NeuralNetwork/include \
	-I./mnist/include

# source files
NNDIR = ./NeuralNetwork/source
SRCS = 	./main.cpp	\
		$(NNDIR)/utils.cpp	\
		$(NNDIR)/Network.cpp

OBJDIR = ./build
OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)

# compile step
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(FLAGS) -c $< -o $@

# link step
$(NAME): $(OBJS)
	$(CC) $(OBJS) -o $(NAME)

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
