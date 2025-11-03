NAME = NeuralNet_CPP
CC = g++
FLAGS = -Wall -Werror -Wextra -std=c++17

SRCDIR = ./srcs
SRCS = 	$(SRCDIR)/main.cpp	\
		$(SRCDIR)/utils.cpp	\
		$(SRCDIR)/Network.cpp

OBJDIR = ./build
OBJS = $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# compile step
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
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
