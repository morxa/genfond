(define (problem blocks-011-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b1) (clear b3) (clear b9) (handempty) (on b0 b2) (on b10 b5) (on b3 b4) (on b4 b8) (on b6 b7) (on b7 b10) (on b8 b6) (ontable b1) (ontable b2) (ontable b5) (ontable b9))
    (:goal (clear b5))
)