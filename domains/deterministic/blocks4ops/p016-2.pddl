(define (problem blocks-016-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b5) (clear b8) (handempty) (on b1 b11) (on b10 b6) (on b11 b10) (on b13 b2) (on b14 b1) (on b15 b14) (on b2 b15) (on b3 b9) (on b4 b12) (on b6 b7) (on b7 b3) (on b8 b13) (on b9 b4) (ontable b0) (ontable b12) (ontable b5))
    (:goal (and (clear b10) (on b10 b5) (on b5 b3) (on b3 b11) (on b11 b6) (on b6 b2) (on b2 b0) (on b0 b8) (on b8 b15) (on b15 b12) (ontable b12)))
)