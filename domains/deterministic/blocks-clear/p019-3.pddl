(define (problem blocks-019-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b15) (clear b16) (clear b4) (clear b7) (clear b8) (handempty) (on b0 b13) (on b1 b17) (on b11 b3) (on b12 b14) (on b13 b9) (on b14 b6) (on b15 b5) (on b17 b10) (on b18 b2) (on b3 b1) (on b6 b11) (on b7 b18) (on b9 b12) (ontable b10) (ontable b16) (ontable b2) (ontable b4) (ontable b5) (ontable b8))
    (:goal (clear b6))
)