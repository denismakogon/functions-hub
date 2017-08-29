FROM funcy/go:dev
RUN mkdir -p $GOPATH/src/github.com/denismakogon/functions-hub/hot/env
ADD . $GOPATH/src/github.com/denismakogon/functions-hub/hot/env
WORKDIR $GOPATH/src/github.com/denismakogon/functions-hub/hot/env
RUN go build -o func
ENTRYPOINT ["./func"]
