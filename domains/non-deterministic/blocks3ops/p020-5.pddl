(define (problem blocks-020-5)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b1) (clear b16) (clear b18) (clear b4) (on b0 b10) (on b1 b8) (on b10 b14) (on b12 b11) (on b13 b7) (on b14 b5) (on b16 b2) (on b18 b13) (on b19 b15) (on b2 b6) (on b5 b12) (on b6 b9) (on b7 b17) (on b8 b19) (on b9 b3) (ontable b11) (ontable b15) (ontable b17) (ontable b3) (ontable b4))
    (:goal (and (clear b16) (on b16 b17) (on b17 b13) (on b13 b4) (on b4 b15) (on b15 b14) (on b14 b10) (ontable b10)))
)