CC=g++
CFLAGS= -std=c++11 
DEBUG_FLAGS= -ggdb -g
OPTIMIZE_FLAG= -O2
LIB_FLAGS= -I/usr/include/python2.7
LIB= -lpython2.7
OBJECTS= data_source.o
PROG= dml.exe
PROG_DEBUG= dml_debug.exe
TEST= test.exe

$(PROG) : $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@

%.o: %.cpp matrix.hpp dmlpack.h data_source.cpp
	$(CC) $(OPTIMIZE_FLAG) $(DEBUG_FLAGS) $(LIB_FLAGS) $(LIB) $(CFLAGS)  -c $<

debug: 
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@

clean:
	rm $(OBJECTS) *.gch *.exe
run:
	./$(PROG_DEBUG)

test : $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB_FLAGS) $(LIB)  $(DEBUG_FLAGS) $(CFLAGS) test.cpp -o $(TEST)


test_dis : $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB_FLAGS) $(LIB) $(CFLAGS) -S -c test.cpp
