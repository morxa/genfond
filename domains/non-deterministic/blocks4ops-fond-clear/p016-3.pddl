(define (problem blocks-016-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b7) (clear b8) (handempty) (on b0 b14) (on b1 b11) (on b10 b12) (on b11 b10) (on b12 b2) (on b13 b6) (on b15 b3) (on b2 b4) (on b3 b5) (on b4 b13) (on b5 b9) (on b7 b15) (on b9 b1) (ontable b14) (ontable b6) (ontable b8))
    (:goal (clear b2))
)