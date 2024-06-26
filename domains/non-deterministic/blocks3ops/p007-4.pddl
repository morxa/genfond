(define (problem blocks-007-4)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b2 b3 b4 b5 b6)
    (:init (clear b1) (clear b2) (clear b3) (clear b4) (on b3 b6) (on b4 b5) (on b5 b0) (ontable b0) (ontable b1) (ontable b2) (ontable b6))
    (:goal (and (clear b2) (on b2 b1) (ontable b1)))
)