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
digit(37).
digit(38).
digit(39).
digit(40).
digit(41).
digit(42).
digit(43).
digit(44).
digit(45).
digit(46).
digit(47).
digit(48).
digit(49).
digit(50).
digit(51).
digit(52).
digit(53).
digit(54).
digit(55).
digit(56).
digit(57).
digit(58).
digit(59).
digit(60).
digit(61).
digit(62).
digit(63).
digit(64).
digit(65).
digit(66).
digit(67).
digit(68).
digit(69).
digit(70).
digit(71).
digit(72).
digit(73).
digit(74).
digit(75).
digit(76).
digit(77).
digit(78).
digit(79).
digit(80).
digit(81).
digit(82).
digit(83).
digit(84).
digit(85).
digit(86).
digit(87).
digit(88).
digit(89).
digit(90).
digit(91).
digit(92).
digit(93).
digit(94).
digit(95).
digit(96).
digit(97).
digit(98).
digit(99).
digit(100).
digit(101).
digit(102).
digit(103).
digit(104).
digit(105).
digit(106).
digit(107).
digit(108).
digit(109).
digit(110).
digit(111).
digit(112).
digit(113).
digit(114).
digit(115).
digit(116).
digit(117).
digit(118).
digit(119).
digit(120).
digit(121).
digit(122).
digit(123).
digit(124).
digit(125).
digit(126).
digit(127).
digit(128).
digit(129).
digit(130).
digit(131).
digit(132).
digit(133).
digit(134).
digit(135).
digit(136).
digit(137).
digit(138).
digit(139).
digit(140).
digit(141).
digit(142).
digit(143).
digit(144).
digit(145).

eq(X, X).

% Addition directly on the embeddings
sum(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, Z) :- 
    sum_emb(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, Z).

% Add two lists of embeddings (input1, input2, result)
sum_emb([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []).
sum_emb([~H1], [~H2], [~H3], [~H4], [~H5], [~H6], [~H7], [~H8], [~H9], [~H10], [~H11], [~H12], [~H13], [~H14], [~H15], [~H16], [HZ]) :-
    digit(HZ),
    eq(~HZ, ~plus16(H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16)).


% for digit eval:
mnist(X, N) :- digit(N), eq(X, ~N).