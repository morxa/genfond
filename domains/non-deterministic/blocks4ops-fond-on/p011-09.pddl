(define (problem blocks-011-09)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b2) (clear b3) (clear b7) (clear b9) (handempty) (on b0 b6) (on b1 b0) (on b4 b10) (on b6 b8) (on b8 b4) (on b9 b5) (ontable b10) (ontable b2) (ontable b3) (ontable b5) (ontable b7))
    (:goal (on b10 b5))
)