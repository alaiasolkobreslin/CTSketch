% MNIST sum:
digit(0).
digit(1).
digit(2).
digit(3).
digit(4).
digit(5).
digit(6).
digit(7).
digit(8).
digit(9).
digit(10).
digit(11).
digit(12).
digit(13).
digit(14).
digit(15).
digit(16).
digit(17).
digit(18).

eq(X, X).

% Addition directly on the embeddings
sum(X,Y,Z) :- sum_emb(X, Y, Z).

% Add two lists of embeddings (input1, input2, result)
sum_emb([], [], []).
sum_emb([~HX], [~HY], [HZ]) :-
    digit(HZ),
    eq(~HZ, ~plus(HX,HY)).


% for digit eval:
mnist(X, N) :- digit(N), eq(X, ~N).