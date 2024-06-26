(define (problem blocks-011-1)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b4) (clear b7) (on b0 b3) (on b10 b8) (on b2 b0) (on b3 b9) (on b4 b2) (on b7 b10) (on b8 b5) (on b9 b6) (ontable b1) (ontable b5) (ontable b6))
    (:goal (and (clear b8) (on b8 b9) (on b9 b4) (ontable b4)))
)