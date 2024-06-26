(define (problem blocks-017-2)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b11) (clear b14) (clear b15) (clear b3) (clear b5) (on b0 b10) (on b10 b9) (on b11 b2) (on b12 b1) (on b13 b7) (on b14 b4) (on b15 b12) (on b3 b6) (on b4 b0) (on b6 b8) (on b7 b16) (on b8 b13) (ontable b1) (ontable b16) (ontable b2) (ontable b5) (ontable b9))
    (:goal (and (clear b1) (on b1 b2) (on b2 b13) (on b13 b8) (on b8 b5) (on b5 b0) (on b0 b16) (on b16 b10) (on b10 b6) (on b6 b12) (on b12 b4) (on b4 b3) (on b3 b15) (on b15 b7) (on b7 b14) (on b14 b9) (on b9 b11) (ontable b11)))
)