(define (problem blocks-016-05)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b10) (clear b13) (clear b14) (clear b8) (handempty) (on b0 b11) (on b1 b15) (on b11 b2) (on b12 b6) (on b14 b4) (on b3 b12) (on b4 b9) (on b5 b7) (on b6 b1) (on b7 b3) (on b8 b0) (on b9 b5) (ontable b10) (ontable b13) (ontable b15) (ontable b2))
    (:goal (on b15 b2))
)