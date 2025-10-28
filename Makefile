NAME = NeuralNet_C
CC = gcc
FLAGS = -Wall -Werror -Wextra

SRCDIR = ./srcs
SRCS = $(SRCDIR)/main.c

OBJDIR = ./build
OBJS = $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# compile step
$(OBJDIR)/%.o: $(SRCDIR)/%.c
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
