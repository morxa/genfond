(define (problem blocks-016-04)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b10) (clear b13) (clear b15) (clear b6) (handempty) (on b0 b9) (on b1 b14) (on b10 b2) (on b11 b8) (on b12 b0) (on b14 b7) (on b15 b5) (on b2 b4) (on b5 b3) (on b6 b12) (on b7 b11) (ontable b13) (ontable b3) (ontable b4) (ontable b8) (ontable b9))
    (:goal (on b3 b11))
)