(define (problem blocks-018-1)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b11) (clear b3) (clear b4) (clear b5) (clear b8) (on b0 b2) (on b1 b17) (on b10 b6) (on b11 b1) (on b12 b0) (on b13 b16) (on b14 b13) (on b15 b10) (on b16 b7) (on b2 b9) (on b4 b12) (on b6 b14) (on b9 b15) (ontable b17) (ontable b3) (ontable b5) (ontable b7) (ontable b8))
    (:goal (and (clear b13) (on b13 b10) (on b10 b7) (on b7 b14) (on b14 b2) (on b2 b5) (on b5 b8) (on b8 b3) (on b3 b11) (on b11 b6) (on b6 b1) (on b1 b15) (on b15 b16) (on b16 b4) (on b4 b17) (ontable b17)))
)