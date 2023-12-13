(define (problem tireworld-089-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l9 - location)
    (:init (road l0 l25) (road l0 l6) (road l1 l30) (road l1 l70) (road l10 l40) (road l10 l66) (road l11 l12) (road l11 l55) (road l11 l68) (road l12 l11) (road l12 l16) (road l12 l6) (road l13 l27) (road l13 l88) (road l14 l7) (road l14 l77) (road l15 l44) (road l15 l52) (road l16 l12) (road l16 l46) (road l17 l53) (road l17 l62) (road l18 l23) (road l18 l33) (road l19 l61) (road l19 l9) (road l2 l31) (road l2 l58) (road l20 l3) (road l20 l34) (road l21 l32) (road l21 l82) (road l22 l51) (road l22 l71) (road l23 l18) (road l23 l35) (road l24 l60) (road l24 l77) (road l25 l0) (road l25 l62) (road l25 l74) (road l26 l85) (road l26 l86) (road l27 l13) (road l27 l29) (road l28 l57) (road l28 l59) (road l29 l27) (road l29 l86) (road l3 l20) (road l3 l60) (road l30 l1) (road l30 l7) (road l31 l2) (road l31 l60) (road l32 l21) (road l32 l60) (road l33 l18) (road l33 l69) (road l34 l20) (road l34 l37) (road l35 l23) (road l35 l78) (road l36 l4) (road l36 l45) (road l37 l34) (road l37 l56) (road l38 l51) (road l38 l8) (road l39 l45) (road l39 l63) (road l4 l36) (road l4 l49) (road l40 l10) (road l40 l80) (road l41 l48) (road l41 l77) (road l42 l47) (road l42 l48) (road l43 l64) (road l43 l80) (road l44 l15) (road l44 l87) (road l45 l36) (road l45 l39) (road l46 l16) (road l46 l47) (road l47 l42) (road l47 l46) (road l48 l41) (road l48 l42) (road l48 l70) (road l49 l4) (road l49 l61) (road l5 l75) (road l50 l68) (road l50 l73) (road l51 l22) (road l51 l38) (road l51 l59) (road l52 l15) (road l52 l65) (road l53 l17) (road l53 l83) (road l54 l66) (road l54 l7) (road l54 l78) (road l55 l11) (road l55 l57) (road l55 l7) (road l56 l37) (road l56 l76) (road l57 l28) (road l57 l55) (road l58 l2) (road l58 l8) (road l59 l28) (road l59 l51) (road l6 l0) (road l6 l12) (road l60 l24) (road l60 l3) (road l60 l31) (road l60 l32) (road l61 l19) (road l61 l49) (road l62 l17) (road l62 l25) (road l63 l39) (road l63 l72) (road l64 l43) (road l64 l84) (road l65 l52) (road l65 l85) (road l66 l10) (road l66 l54) (road l67 l73) (road l67 l79) (road l68 l11) (road l68 l50) (road l69 l33) (road l69 l70) (road l7 l14) (road l7 l30) (road l7 l54) (road l7 l55) (road l70 l1) (road l70 l48) (road l70 l69) (road l71 l22) (road l71 l84) (road l72 l63) (road l72 l88) (road l73 l50) (road l73 l67) (road l74 l25) (road l74 l76) (road l75 l5) (road l75 l8) (road l75 l83) (road l76 l56) (road l76 l74) (road l77 l14) (road l77 l24) (road l77 l41) (road l78 l35) (road l78 l54) (road l79 l67) (road l79 l87) (road l8 l38) (road l8 l58) (road l8 l75) (road l80 l40) (road l80 l43) (road l81 l82) (road l81 l9) (road l82 l21) (road l82 l81) (road l83 l53) (road l83 l75) (road l84 l64) (road l84 l71) (road l85 l26) (road l85 l65) (road l86 l26) (road l86 l29) (road l87 l44) (road l87 l79) (road l88 l13) (road l88 l72) (road l9 l19) (road l9 l81) (spare-in l13) (spare-in l15) (spare-in l17) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l3) (spare-in l34) (spare-in l37) (spare-in l39) (spare-in l41) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l51) (spare-in l53) (spare-in l56) (spare-in l57) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l65) (spare-in l69) (spare-in l71) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l81) (spare-in l83) (spare-in l85) (spare-in l9) (vehicle-at l70))
    (:goal (vehicle-at l5))
)