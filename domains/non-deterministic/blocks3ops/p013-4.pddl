(define (problem blocks-013-4)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b5) (on b0 b9) (on b1 b3) (on b10 b2) (on b11 b4) (on b12 b8) (on b3 b10) (on b4 b0) (on b5 b12) (on b6 b1) (on b7 b11) (on b8 b7) (on b9 b6) (ontable b2))
    (:goal (and (clear b5) (on b5 b3) (on b3 b11) (on b11 b4) (on b4 b9) (on b9 b8) (on b8 b7) (on b7 b10) (on b10 b2) (on b2 b12) (ontable b12)))
)