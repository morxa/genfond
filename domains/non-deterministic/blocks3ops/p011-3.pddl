(define (problem blocks-011-3)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b10) (clear b4) (on b0 b2) (on b1 b3) (on b10 b5) (on b2 b9) (on b3 b7) (on b4 b8) (on b5 b1) (on b8 b0) (on b9 b6) (ontable b6) (ontable b7))
    (:goal (and (clear b5) (on b5 b7) (on b7 b3) (on b3 b10) (on b10 b2) (on b2 b8) (on b8 b0) (on b0 b1) (ontable b1)))
)