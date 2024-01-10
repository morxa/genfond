(define (problem blocks-015-1)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b11) (clear b13) (clear b2) (clear b6) (on b0 b5) (on b10 b3) (on b12 b14) (on b14 b7) (on b2 b4) (on b4 b0) (on b5 b9) (on b7 b10) (on b8 b12) (on b9 b8) (ontable b1) (ontable b11) (ontable b13) (ontable b3) (ontable b6))
    (:goal (and (clear b0) (on b0 b5) (on b5 b10) (on b10 b6) (on b6 b2) (on b2 b3) (on b3 b7) (on b7 b11) (on b11 b1) (on b1 b12) (on b12 b13) (on b13 b8) (on b8 b4) (on b4 b9) (on b9 b14) (ontable b14)))
)