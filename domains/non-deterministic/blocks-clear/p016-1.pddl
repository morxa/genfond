(define (problem blocks-016-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b11) (clear b12) (clear b14) (clear b5) (clear b8) (clear b9) (handempty) (on b10 b15) (on b11 b2) (on b12 b0) (on b13 b7) (on b2 b3) (on b3 b6) (on b4 b10) (on b8 b13) (on b9 b4) (ontable b0) (ontable b1) (ontable b14) (ontable b15) (ontable b5) (ontable b6) (ontable b7))
    (:goal (clear b6))
)