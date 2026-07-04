# sml-safetensors build
MLTON      ?= mlton
BIN        := bin
LIBDIR     := lib/github.com/sjqtentacles/sml-safetensors
VENDOR     := lib/github.com/sjqtentacles
TEST_MLB   := test/sources.mlb
SRCS       := $(wildcard $(LIBDIR)/*.sml $(LIBDIR)/*.sig) \
              $(wildcard $(VENDOR)/sml-json/*.sml $(VENDOR)/sml-json/*.sig) \
              $(wildcard $(VENDOR)/sml-parsec/*.sml $(VENDOR)/sml-parsec/*.sig) \
              $(wildcard $(VENDOR)/sml-float/*.sml $(VENDOR)/sml-float/*.sig) \
              $(wildcard test/*.sml) $(TEST_MLB) \
              $(LIBDIR)/sources.mlb $(VENDOR)/sml-json/sources.mlb \
              $(VENDOR)/sml-float/sources.mlb $(VENDOR)/sml-parsec/parsec.mlb

.PHONY: all test poly test-poly all-tests clean

all: $(BIN)/test-mlton

$(BIN)/test-mlton: $(SRCS) | $(BIN)
	$(MLTON) -output $@ $(TEST_MLB)

test: $(BIN)/test-mlton
	$(BIN)/test-mlton

poly: $(BIN)/test-poly

$(BIN)/test-poly: $(SRCS) tools/polybuild | $(BIN)
	sh tools/polybuild -o $@ $(TEST_MLB)

test-poly: $(BIN)/test-poly
	$(BIN)/test-poly

all-tests: test test-poly

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
