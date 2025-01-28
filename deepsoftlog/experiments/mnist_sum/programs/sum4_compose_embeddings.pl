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
digit(19).
digit(20).
digit(21).
digit(22).
digit(23).
digit(24).
digit(25).
digit(26).
digit(27).
digit(28).
digit(29).
digit(30).
digit(31).
digit(32).
digit(33).
digit(34).
digit(35).
digit(36).

eq(X, X).

% Addition directly on the embeddings
sum(N1, N2, N3, N4, Z) :- sum_emb(N1, N2, T1), cut(HERE), sum_emb(N3, N4, T2), cut(HERE), sum2_emb(T1, T2, Z).

% Add two lists of embeddings (input1, input2, result)
sum_emb([], [], []).
sum_emb([~H1], [~H2],[HZ]) :-
    digit(HZ),
    eq(~HZ, ~plus(H1,H2)).

% Add two lists of embeddings (input1, input2, result)
sum2_emb([], [], []).
sum2_emb([~H1], [~H2],[HZ]) :-
    digit(HZ),
    eq(~HZ, ~plus(H1,H2)).

% for digit eval:
mnist(X, N) :- digit(N), eq(X, ~N).