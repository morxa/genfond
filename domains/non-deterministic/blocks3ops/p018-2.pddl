(define (problem blocks-018-2)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b13) (clear b3) (clear b4) (on b0 b2) (on b10 b5) (on b11 b8) (on b12 b9) (on b13 b17) (on b14 b1) (on b16 b11) (on b17 b7) (on b3 b6) (on b4 b12) (on b5 b0) (on b6 b16) (on b7 b15) (on b8 b14) (on b9 b10) (ontable b1) (ontable b15) (ontable b2))
    (:goal (and (clear b8) (on b8 b1) (on b1 b10) (on b10 b13) (on b13 b12) (on b12 b6) (on b6 b15) (on b15 b16) (on b16 b17) (ontable b17)))
)