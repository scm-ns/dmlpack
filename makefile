CC=g++
CFLAGS= -std=c++11 
DEBUG_FLAGS= -ggdb
OPTIMIZE_FLAG= 
LIB_FLAGS= -I/usr/include/python2.7
LIB= -lpython2.7
OBJECTS= data_source.o
PROG= dml.exe
PROG_DEBUG= dml_debug.exe

$(PROG) : $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@

%.o: %.cpp matrix.h dmlpack.h data_source.cpp
	$(CC) $(OPTIMIZE_FLAG) $(DEBUG_FLAGS) $(LIB_FLAGS) $(LIB) $(CFLAGS) -c $<

debug: 
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@

clean:
	rm $(PROG_DEBUG) $(OBJECTS) *.gch test
run:
	./$(PROG_DEBUG)

test: $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB_FLAGS) $(LIB) $(CFLAGS) test.cpp -o $@
